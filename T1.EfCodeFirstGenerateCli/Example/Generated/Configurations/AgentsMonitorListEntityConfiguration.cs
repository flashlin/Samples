using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class AgentsMonitorListEntityConfiguration : IEntityTypeConfiguration<AgentsMonitorListEntity>
    {
        public void Configure(EntityTypeBuilder<AgentsMonitorListEntity> builder)
        {
            builder.ToTable("AgentsMonitorList");

            builder.HasKey(x => new { x.ParentID, x.ChildID });

            builder.Property(x => x.ParentID)
                .HasColumnType("int(10,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.ChildID)
                .HasColumnType("int(10,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.ModifiedDate)
                .HasColumnType("datetime")
            ;

        }
    }
}
