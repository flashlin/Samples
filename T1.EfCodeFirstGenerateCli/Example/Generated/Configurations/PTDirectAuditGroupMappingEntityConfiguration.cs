using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class PTDirectAuditGroupMappingEntityConfiguration : IEntityTypeConfiguration<PTDirectAuditGroupMappingEntity>
    {
        public void Configure(EntityTypeBuilder<PTDirectAuditGroupMappingEntity> builder)
        {
            builder.ToTable("PTDirectAuditGroupMapping");


            builder.Property(x => x.AuditGroupId)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.AuditGroupName)
                .HasColumnType("varchar(50)")
                .IsRequired()
                .HasMaxLength(50)
            ;

        }
    }
}
