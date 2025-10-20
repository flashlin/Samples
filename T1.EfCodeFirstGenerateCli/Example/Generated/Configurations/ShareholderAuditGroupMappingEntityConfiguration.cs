using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class ShareholderAuditGroupMappingEntityConfiguration : IEntityTypeConfiguration<ShareholderAuditGroupMappingEntity>
    {
        public void Configure(EntityTypeBuilder<ShareholderAuditGroupMappingEntity> builder)
        {
            builder.ToTable("ShareholderAuditGroupMapping");


            builder.Property(x => x.CustomerId)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

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
